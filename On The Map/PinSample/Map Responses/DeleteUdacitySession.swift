//
//  DeleteUdacitySession.swift
//  PinSample
//
//  Created by Ebraham Alskaf on 08/06/2024.
//  Copyright © 2024 Udacity. All rights reserved.
//

import Foundation

struct Session: Codable {
    let id: String
    let expiration: String
}

struct DeleteUdacitySession: Codable {
    let session: Session
    
    enum CodingKeys: String, CodingKey {
        case session
    }
}
